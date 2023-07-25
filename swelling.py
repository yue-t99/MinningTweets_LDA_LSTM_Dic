from Scweet.scweet import scrape
#from Scweet.user import get_user_information, get_users_following, get_users_followers

data = scrape(words=['\"Self-driving cars\"','\"selfdriving cars\"','\"self driving\"',
                     '\"Autonomous Cars\"','\"autonomous cars\"','\"autonomous car\"',
                     '\"Automated Vehicles\"','\"autonomous vehicles\"','\"Tesla car\"',
                     '\"Self-drivingcars\"','\"selfdrivingcars\"','\"selfdriving\"',
                     '\"AutonomousCars\"','\"autonomouscars\"','\"autonomouscar\"',
                     '\"AutomatedVehicles\"','\"autonomousvehicles\"','\"Teslacar\"'
                     ], since="2023-1-1", until="2023-3-31", from_account = None, 
              interval=1, headless=False, display_type="Top", save_images=False, lang="en",
              resume=False, filter_replies=False, proximity=False)

print("success")